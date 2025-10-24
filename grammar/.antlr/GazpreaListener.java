// Generated from /home/chloe/School/CMPUT415/gazprea-sweaties/grammar/Gazprea.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link GazpreaParser}.
 */
public interface GazpreaListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link GazpreaParser#file}.
	 * @param ctx the parse tree
	 */
	void enterFile(GazpreaParser.FileContext ctx);
	/**
	 * Exit a parse tree produced by {@link GazpreaParser#file}.
	 * @param ctx the parse tree
	 */
	void exitFile(GazpreaParser.FileContext ctx);
}