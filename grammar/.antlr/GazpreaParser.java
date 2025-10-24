// Generated from /home/chloe/School/CMPUT415/gazprea-sweaties/grammar/Gazprea.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class GazpreaParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		AND=1, AS=2, BOOLEAN=3, BREAK=4, BY=5, CALL=6, CHARACTER=7, COLUMNS=8, 
		CONST=9, CONTINUE=10, ELSE=11, FALSE=12, FORMAT=13, FUNCTION=14, IF=15, 
		IN=16, INTEGER=17, LENGTH=18, LOOP=19, NOT=20, OR=21, PROCEDURE=22, REAL=23, 
		RETURN=24, RETURNS=25, REVERSE=26, ROWS=27, STD_INPUT=28, STD_OUTPUT=29, 
		STREAM_STATE=30, STRING=31, TRUE=32, TUPLE=33, TYPEALIAS=34, VAR=35, VECTOR=36, 
		WHILE=37, XOR=38, ID=39, SL_COMMENT=40, ML_COMMENT=41, WS=42;
	public static final int
		RULE_file = 0;
	private static String[] makeRuleNames() {
		return new String[] {
			"file"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'and'", "'as'", "'boolean'", "'break'", "'by'", "'call'", "'character'", 
			"'columns'", "'const'", "'continue'", "'else'", "'false'", "'format'", 
			"'function'", "'if'", "'in'", "'integer'", "'length'", "'loop'", "'not'", 
			"'or'", "'procedure'", "'real'", "'return'", "'returns'", "'reverse'", 
			"'rows'", "'std_input'", "'std_output'", "'stream_state'", "'string'", 
			"'true'", "'tuple'", "'typealias'", "'var'", "'vector'", "'while'", "'xor'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "AND", "AS", "BOOLEAN", "BREAK", "BY", "CALL", "CHARACTER", "COLUMNS", 
			"CONST", "CONTINUE", "ELSE", "FALSE", "FORMAT", "FUNCTION", "IF", "IN", 
			"INTEGER", "LENGTH", "LOOP", "NOT", "OR", "PROCEDURE", "REAL", "RETURN", 
			"RETURNS", "REVERSE", "ROWS", "STD_INPUT", "STD_OUTPUT", "STREAM_STATE", 
			"STRING", "TRUE", "TUPLE", "TYPEALIAS", "VAR", "VECTOR", "WHILE", "XOR", 
			"ID", "SL_COMMENT", "ML_COMMENT", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "Gazprea.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public GazpreaParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FileContext extends ParserRuleContext {
		public TerminalNode EOF() { return getToken(GazpreaParser.EOF, 0); }
		public FileContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_file; }
	}

	public final FileContext file() throws RecognitionException {
		FileContext _localctx = new FileContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_file);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(5);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,0,_ctx);
			while ( _alt!=1 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1+1 ) {
					{
					{
					setState(2);
					matchWildcard();
					}
					} 
				}
				setState(7);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,0,_ctx);
			}
			setState(8);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\u0004\u0001*\u000b\u0002\u0000\u0007\u0000\u0001\u0000\u0005\u0000\u0004"+
		"\b\u0000\n\u0000\f\u0000\u0007\t\u0000\u0001\u0000\u0001\u0000\u0001\u0000"+
		"\u0001\u0005\u0000\u0001\u0000\u0000\u0000\n\u0000\u0005\u0001\u0000\u0000"+
		"\u0000\u0002\u0004\t\u0000\u0000\u0000\u0003\u0002\u0001\u0000\u0000\u0000"+
		"\u0004\u0007\u0001\u0000\u0000\u0000\u0005\u0006\u0001\u0000\u0000\u0000"+
		"\u0005\u0003\u0001\u0000\u0000\u0000\u0006\b\u0001\u0000\u0000\u0000\u0007"+
		"\u0005\u0001\u0000\u0000\u0000\b\t\u0005\u0000\u0000\u0001\t\u0001\u0001"+
		"\u0000\u0000\u0000\u0001\u0005";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}